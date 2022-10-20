{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Data.Accelerate.TensorFlow.Operation where

import Data.Array.Accelerate.AST.Exp
import Data.Array.Accelerate.AST.Operation
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Analysis.Hash.Exp
import Data.Array.Accelerate.Analysis.Hash.Operation
import Data.Array.Accelerate.Backend
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Graph hiding (Var)
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Labels
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.Representation.Shape (shapeType)
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.AST.Environment

data TensorOp op where
  TConst :: ScalarType s -> s -> TensorOp (Out sh s -> ())
  TPrimFun :: PrimFun (a -> b) -> TensorOp (In sh a -> Out sh b -> ())

instance PrettyOp TensorOp where
  prettyOp (TConst _ _) = "TTensor"
  prettyOp (TPrimFun _) = "TBinOp"

instance NFData' TensorOp where
  rnf' !_ = ()

mapXTimesTwoPlusOne :: forall env sh. Arg env (In sh Int64) -> Arg env (Out sh Int64) -> OperationAcc TensorOp env ()
mapXTimesTwoPlusOne
  (
    ArgArray _ arrayR@(ArrayR sh _)
    gvIn
    gvbIn
  )
  argOut@(
    ArgArray _ _
    gvOut
    gvbOut
  )
  = let
      tInt64 :: NumType Int64
      tInt64 = IntegralNumType TypeInt64

      timesOp = TPrimFun (PrimMul tInt64)
      plusOp  = TPrimFun (PrimAdd tInt64) -- Todo: include

      tNewIn :: TupR ScalarType (Int64, Int64)
      tNewIn = TupRpair
        (TupRsingle (SingleScalarType (NumSingleType tInt64)))
        (TupRsingle (SingleScalarType (NumSingleType tInt64)))

      -- getNewGvbIn :: GroundVars env (Buffers Int64) -> GroundVars env (Buffers (Int64, Int64))
      -- getNewGvbIn TupRunit = Alet 
      -- getNewGvbIn (TupRsingle _) = _
      -- getNewGvbIn (TupRpair _ _) = _

      tNewOut :: TupR ScalarType Int64
      tNewOut = TupRsingle (SingleScalarType (NumSingleType tInt64))

      newGvbOut :: TupR (Var GroundR env) (Buffer Int64)
      newGvbOut = undefined

      int64Buffer :: GroundR (Buffer Int64)
      int64Buffer = GroundRbuffer $ SingleScalarType (NumSingleType tInt64)

      gvIn' :: TupR (Var GroundR (env, Buffer Int64)) sh
      gvIn' = mapTupR (weaken (weakenSucc weakenId)) gvIn

    -- in Exec timesOp (
    --   ArgArray In (ArrayR shIn tNewIn) gvIn (getNewGvbIn gvbIn)
    --   :>: ArgArray Out (ArrayR shOut tNewOut) gvOut newGvbOut
    --   :>: ArgsNil
    -- )
    in -- eerst nieuwe buffer aanmaken, eerst array aanmaken van zelfde grootte
       Alet -- kan je gebruiken voor nieuwe variabelen of side effects uitvoeren en dan doorgaan met iets anders
        (LeftHandSideSingle int64Buffer) -- variable introduceren
        (TupRsingle Unique) -- uniqueness van nieuwe variabele
        (Alloc sh (SingleScalarType (NumSingleType tInt64)) (groundToExpVar (shapeType sh) gvIn))
        -- array vullen met tweeën
        $ Alet (LeftHandSideWildcard TupRunit)
          TupRunit 
          (Exec 
            (TConst (SingleScalarType (NumSingleType tInt64)) 2) 
            (ArgArray Out arrayR gvIn' (TupRsingle $ Var int64Buffer ZeroIdx) :>: ArgsNil) -- (TupRsingle $ Var int64Buffer ZeroIdx) refereert naar een array
          ) 
          -- keer twee
          $ Exec 
            (TPrimFun (PrimMul tInt64)) 
            (ArgArray In (ArrayR sh tNewIn) gvIn' (TupRpair (mapTupR (weaken (weakenSucc weakenId)) gvbIn) (TupRsingle $ Var int64Buffer ZeroIdx)) :>: weaken (weakenSucc weakenId) argOut :>: ArgsNil) 

-- voor het dynamisch toevoegen van nieuwe variabelen: zie Trafo/Vars.hs => declareVars, gebruik pattern matching in guards
-- e.g.  | DeclareVars lhs w k <- declareVars $ buffersR (TupRpair ty1 ty2) =

-- mkMapF :: Arg env (Fun' (s -> t)) -> Arg env (In sh s) -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
-- Constant values
-- mkMapF (ArgFun (Body (Const s e)))   _ _ = Exec (TConst s e) ArgsNil
-- mkMapF (ArgFun (Body (PrimConst _))) _ _ = undefined
-- -- Primitive scalar operations
-- mkMapF (ArgFun (Body (PrimApp f exp))) (ArgArray m (ArrayR sh tR) g b) o
--   = Exec (TPrimFun f) (ArgArray m (ArrayR sh _) g _ :>: _ :>: ArgsNil)

-- mkMapF (ArgFun (Body (ArrayInstr _ _))) input output = undefined

-- Others
-- mkMapF (ArgFun (Body (Let _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (Evar _))) input output = undefined
-- mkMapF (ArgFun (Body (Foreign _ _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (VecUnpack _ _))) input output = undefined
-- mkMapF (ArgFun (Body (IndexSlice _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (IndexFull _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (FromIndex _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (Case _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (Cond _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (While _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (Undef _))) input output = undefined
-- mkMapF (ArgFun (Body (Coerce _ _ _))) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSideSingle (SingleScalarType (NumSingleType (IntegralNumType _)))) f)) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSideSingle (SingleScalarType (NumSingleType (FloatingNumType _)))) f)) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSideSingle (VectorScalarType _)) f)) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSideWildcard _) f)) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSidePair _ _) f)) input output = undefined

-- -- Redundant pattern matches
-- mkMapF (ArgFun (Body (Pair _ _))) input output = undefined
-- mkMapF (ArgFun (Body Nil)) input output = undefined
-- mkMapF (ArgFun (Body (VecPack _ _))) input output = undefined
-- mkMapF (ArgFun (Body (ShapeSize _ _))) input output = undefined
-- mkMapF (ArgFun (Body (ToIndex _ _ _))) input output = undefined

instance DesugarAcc TensorOp where
  mkMap = undefined
  mkGenerate = undefined
  mkPermute = undefined
