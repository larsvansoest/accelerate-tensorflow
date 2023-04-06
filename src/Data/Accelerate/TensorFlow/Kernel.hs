{-# LANGUAGE GADTs             #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE TypeOperators     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module Data.Accelerate.TensorFlow.Kernel where
import Data.Accelerate.TensorFlow.Type
    ( TFOrd, OneOf, TFNum, TFNum', TFAll, TFFloat, TensorType, TFInt )
import Data.Array.Accelerate.AST.Schedule.Uniform
    ( PrimBool,
      Var(Var),
      PreArgs((:>:)),
      GroundVars,
      Out,
      In,
      Arg(ArgArray),
      NFData'(..),
      Args,
      BaseVars,
      BaseVar,
      Cluster,
      BaseR(BaseRground),
      AccessGroundR )
import Data.Array.Accelerate.Array.Buffer ( Buffer, Buffers )
import Data.Array.Accelerate.Representation.Type
    ( Distributes(reprIsSingle),
      TupR(TupRpair, TupRunit, TupRsingle),
      TypeR )
import Data.Array.Accelerate.Backend
    ( IsKernel(KernelMetadata, compileKernel, KernelOperation),
      PrettyKernel(..),
      PrettyKernelStyle(PrettyKernelBody) )
import Data.Accelerate.TensorFlow.Operation ( TensorOp(..) )
import Data.Array.Accelerate.AST.Kernel ( NoKernelMetadata )
import Data.Array.Accelerate.AST.Environment ( Env )
import Data.Array.Accelerate.Eval
    ( clusterOperations,
      ApplyOperation(ApplyOperation),
      ClusterOperations(ClusterOperations) )
import Data.Array.Accelerate.Analysis.Match ( type (:~:)(..) )
import Data.Array.Accelerate.AST.LeftHandSide
    ( LeftHandSide(LeftHandSidePair, LeftHandSideWildcard) )
import Data.Array.Accelerate.Representation.Array
    ( ArrayR(ArrayR) )
import Data.Array.Accelerate.Type ( scalarTypeWord8, ScalarType )


data TensorKernel env where
  TensorConstant :: TensorType a => ScalarType a -> a -> BaseVar env (Buffer a) -> TensorKernel env
  TensorId       :: OneOf TFAll a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSelect   :: OneOf TFAll a => BaseVar env (Buffer PrimBool) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env

  -- operators from Num
  TensorAdd  :: OneOf TFNum  a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorMul  :: OneOf TFNum  a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSub  :: OneOf TFNum  a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorNeg  :: OneOf TFNum' a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAbs  :: OneOf TFNum' a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSign :: OneOf TFNum' a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env

  -- operators from Integral
  TensorTruncateDiv :: OneOf TFNum a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorTruncateMod :: OneOf TFNum a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorRealDiv     :: OneOf TFNum a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env

  -- operators from Bits & FiniteBits
  TensorBitwiseAnd :: OneOf TFInt a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorBitwiseOr  :: OneOf TFInt a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorBitwiseXor :: OneOf TFInt a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorInvert     :: OneOf TFInt a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  
  -- operators from Fractional and Floating
  TensorReciprocal :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSin        :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorCos        :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorTan        :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAsin       :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAcos       :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAtan       :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSinh       :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorCosh       :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorTanh       :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAsinh      :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAcosh      :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAtanh      :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorExp        :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorSqrt       :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorLog        :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorPow        :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorLog1p      :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorAtan2      :: OneOf TFFloat a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  
  -- relational and equality operators
  TensorLess         :: OneOf TFOrd a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer PrimBool) -> TensorKernel env
  TensorGreater      :: OneOf TFOrd a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer PrimBool) -> TensorKernel env
  TensorLessEqual    :: OneOf TFOrd a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer PrimBool) -> TensorKernel env
  TensorGreaterEqual :: OneOf TFOrd a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer PrimBool) -> TensorKernel env
  TensorEqual        :: OneOf TFOrd a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer PrimBool) -> TensorKernel env
  TensorNotEqual     :: OneOf TFOrd a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer PrimBool) -> TensorKernel env
  TensorMaximum      :: OneOf TFOrd a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  TensorMinimum      :: OneOf TFOrd a => BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env

  TensorLogicalAnd   :: BaseVar env (Buffer PrimBool) -> BaseVar env (Buffer PrimBool) -> BaseVar env (Buffer PrimBool) -> TensorKernel env
  TensorLogicalOr    :: BaseVar env (Buffer PrimBool) -> BaseVar env (Buffer PrimBool) -> BaseVar env (Buffer PrimBool) -> TensorKernel env
  TensorLogicalNot   :: BaseVar env (Buffer PrimBool) -> BaseVar env (Buffer PrimBool) -> TensorKernel env

  TensorCast :: (TensorType a, TensorType b) => BaseVar env (Buffer a) -> BaseVar env (Buffer b) -> TensorKernel env

instance NFData' TensorKernel where
  rnf' !_  = ()

newtype TensorFlowKernelMetadata f =
  TensorFlowKernelMetadata { kernelArgsSize :: Int }

instance IsKernel TensorKernel where
  type KernelOperation TensorKernel = TensorOp
  type KernelMetadata  TensorKernel = NoKernelMetadata

  compileKernel :: Env AccessGroundR env -> Cluster TensorOp args -> Args env args -> TensorKernel env
  compileKernel _ cluster clusterArgs
    | ClusterOperations _ lhs [ApplyOperation operation args] <- clusterOperations cluster clusterArgs
    , Just Refl <- wildcards lhs
    = compileOperation operation args
  compileKernel _ _ _ = error "impossible, did you use SequentialSchedule?"

wildcards :: LeftHandSide a e env env' -> Maybe (env :~: env')
wildcards (LeftHandSideWildcard _)    = Just Refl
wildcards (LeftHandSidePair lhs lhs')
 | Just Refl <- wildcards lhs
 , Just Refl <- wildcards lhs'        = Just Refl
wildcards _                           = Nothing

instance PrettyKernel TensorKernel where
  prettyKernel = PrettyKernelBody True $ \_ _ -> ""

compileOperation :: TensorOp args -> Args env args -> TensorKernel env
compileOperation (TConstant stIn s) (argIn :>: _)              = compileNullaryOperation1 stIn argIn (TensorConstant stIn s)
compileOperation (TId stIn) (argIn :>: argOut :>: _)           = compileUnaryOperation1 stIn stIn argIn argOut TensorId

compileOperation (TSelect stIn) (argIn :>: argOut :>: _)       = compileTernaryOperation1 scalarTypeWord8 stIn stIn stIn argIn argOut TensorSelect
        
compileOperation (TAdd stIn) (argIn :>: argOut :>: _)          = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorAdd
compileOperation (TMul stIn) (argIn :>: argOut :>: _)          = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorMul
compileOperation (TSub stIn) (argIn :>: argOut :>: _)          = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorSub
compileOperation (TNeg stIn) (argIn :>: argOut :>: _)          = compileUnaryOperation1 stIn stIn argIn argOut TensorNeg
compileOperation (TAbs stIn) (argIn :>: argOut :>: _)          = compileUnaryOperation1 stIn stIn argIn argOut TensorAbs
compileOperation (TSign stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorSign

compileOperation (TTruncateDiv stIn) (argIn :>: argOut :>: _)  = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorTruncateDiv
compileOperation (TTruncateMod stIn) (argIn :>: argOut :>: _)  = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorTruncateMod
compileOperation (TRealDiv stIn) (argIn :>: argOut :>: _)      = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorRealDiv

compileOperation (TBitwiseAnd stIn) (argIn :>: argOut :>: _)   = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorBitwiseAnd
compileOperation (TBitwiseOr stIn) (argIn :>: argOut :>: _)    = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorBitwiseOr
compileOperation (TBitwiseXor stIn) (argIn :>: argOut :>: _)   = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorBitwiseXor
compileOperation (TInvert stIn) (argIn :>: argOut :>: _)       = compileUnaryOperation1 stIn stIn argIn argOut TensorInvert

compileOperation (TReciprocal stIn) (argIn :>: argOut :>: _)   = compileUnaryOperation1 stIn stIn argIn argOut TensorReciprocal
compileOperation (TSin stIn) (argIn :>: argOut :>: _)          = compileUnaryOperation1 stIn stIn argIn argOut TensorSin
compileOperation (TCos stIn) (argIn :>: argOut :>: _)          = compileUnaryOperation1 stIn stIn argIn argOut TensorCos
compileOperation (TTan stIn) (argIn :>: argOut :>: _)          = compileUnaryOperation1 stIn stIn argIn argOut TensorTan
compileOperation (TAsin stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorAsin
compileOperation (TAcos stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorAcos
compileOperation (TAtan stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorAtan
compileOperation (TSinh stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorSinh
compileOperation (TCosh stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorCosh
compileOperation (TTanh stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorTanh
compileOperation (TAsinh stIn) (argIn :>: argOut :>: _)        = compileUnaryOperation1 stIn stIn argIn argOut TensorAsinh
compileOperation (TAcosh stIn) (argIn :>: argOut :>: _)        = compileUnaryOperation1 stIn stIn argIn argOut TensorAcosh
compileOperation (TAtanh stIn) (argIn :>: argOut :>: _)        = compileUnaryOperation1 stIn stIn argIn argOut TensorAtanh
compileOperation (TSqrt stIn) (argIn :>: argOut :>: _)         = compileUnaryOperation1 stIn stIn argIn argOut TensorSqrt
compileOperation (TExp stIn) (argIn :>: argOut :>: _)          = compileUnaryOperation1 stIn stIn argIn argOut TensorExp
compileOperation (TLog stIn) (argIn :>: argOut :>: _)          = compileUnaryOperation1 stIn stIn argIn argOut TensorLog
compileOperation (TLog1p stIn) (argIn :>: argOut :>: _)        = compileUnaryOperation1 stIn stIn argIn argOut TensorLog1p

compileOperation (TAtan2 stIn) (argIn :>: argOut :>: _)        = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorAtan2
compileOperation (TLess stIn) (argIn :>: argOut :>: _)         = compileBinaryOperation1 stIn stIn scalarTypeWord8 argIn argOut TensorLess
compileOperation (TGreater stIn) (argIn :>: argOut :>: _)      = compileBinaryOperation1 stIn stIn scalarTypeWord8 argIn argOut TensorGreater
compileOperation (TLessEqual stIn) (argIn :>: argOut :>: _)    = compileBinaryOperation1 stIn stIn scalarTypeWord8 argIn argOut TensorLessEqual
compileOperation (TGreaterEqual stIn) (argIn :>: argOut :>: _) = compileBinaryOperation1 stIn stIn scalarTypeWord8 argIn argOut TensorGreaterEqual
compileOperation (TEqual stIn) (argIn :>: argOut :>: _)        = compileBinaryOperation1 stIn stIn scalarTypeWord8 argIn argOut TensorEqual
compileOperation (TNotEqual stIn) (argIn :>: argOut :>: _)     = compileBinaryOperation1 stIn stIn scalarTypeWord8 argIn argOut TensorNotEqual
compileOperation (TMaximum stIn) (argIn :>: argOut :>: _)      = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorMaximum
compileOperation (TMinimum stIn) (argIn :>: argOut :>: _)      = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorMinimum

compileOperation TLogicalAnd (argIn :>: argOut :>: _)          = compileBinaryOperation1 scalarTypeWord8 scalarTypeWord8 scalarTypeWord8 argIn argOut TensorLogicalAnd
compileOperation TLogicalOr (argIn :>: argOut :>: _)           = compileBinaryOperation1 scalarTypeWord8 scalarTypeWord8 scalarTypeWord8 argIn argOut TensorLogicalOr
compileOperation TLogicalNot (argIn :>: argOut :>: _)          = compileUnaryOperation1 scalarTypeWord8 scalarTypeWord8 argIn argOut TensorLogicalNot

compileOperation (TCast stIn stOut) (argIn :>: argOut :>: _) = compileUnaryOperation1 stIn stOut argIn argOut TensorCast

compileOperation (TPow stIn) (argIn :>: argOut :>: _) = compileBinaryOperation1 stIn stIn stIn argIn argOut TensorPow
compileOperation (TWhere _) _                         = undefined
compileOperation (TTensorScatter _) _                 = undefined
compileOperation (TBooleanMask _) _                   = undefined

compileNullaryOperation1 :: forall a sh env. ScalarType a -> Arg env (Out sh a) -> (BaseVar env (Buffer a) -> TensorKernel env) -> TensorKernel env
compileNullaryOperation1 stOut (ArgArray _ (ArrayR _ a) _ gvbOut) kernel
 | Refl <- reprIsSingle @ScalarType @a @Buffer stOut
 = kernel (groundToBase a gvbOut)

compileUnaryOperation1 :: forall a b sh env. ScalarType a -> ScalarType b -> Arg env (In sh a) -> Arg env (Out sh b) -> (BaseVar env (Buffer a) -> BaseVar env (Buffer b) -> TensorKernel env) -> TensorKernel env
compileUnaryOperation1 stIn stOut (ArgArray _ (ArrayR _ a) _ gvbIn) (ArgArray _ (ArrayR _ b) _ gvbOut) kernel
  | Refl <- reprIsSingle @ScalarType @a @Buffer stIn
  , Refl <- reprIsSingle @ScalarType @b @Buffer stOut
  = kernel (groundToBase a gvbIn) (groundToBase b gvbOut)

compileBinaryOperation1 :: forall a b c sh env. ScalarType a -> ScalarType b -> ScalarType c -> Arg env (In sh (a, b)) -> Arg env (Out sh c) -> (BaseVar env (Buffer a) -> BaseVar env (Buffer b) -> BaseVar env (Buffer c) -> TensorKernel env) -> TensorKernel env
compileBinaryOperation1 stIn1 stIn2 stOut (ArgArray _ (ArrayR _ (TupRpair a b)) _ (TupRpair gvbIn1 gvbIn2)) (ArgArray _ (ArrayR _ c) _ gvbOut) kernel
 | Refl <- reprIsSingle @ScalarType @a @Buffer stIn1
 , Refl <- reprIsSingle @ScalarType @b @Buffer stIn2
 , Refl <- reprIsSingle @ScalarType @c @Buffer stOut
 = kernel (groundToBase a gvbIn1) (groundToBase b gvbIn2) (groundToBase c gvbOut)
compileBinaryOperation1 _ _ _ _ _ _ = error "impossible"

compileTernaryOperation1 :: forall a b c d sh env. ScalarType a -> ScalarType b -> ScalarType c -> ScalarType d -> Arg env (In sh (a, (b, c))) -> Arg env (Out sh d) -> (BaseVar env (Buffer a) -> BaseVar env (Buffer b) -> BaseVar env (Buffer c) -> BaseVar env (Buffer d) -> TensorKernel env) -> TensorKernel env
compileTernaryOperation1 stIn1 stIn2 stIn3 stOut (ArgArray _ (ArrayR _ (TupRpair a (TupRpair b c))) _ (TupRpair gvbIn1 (TupRpair gvbIn2 gvbIn3))) (ArgArray _ (ArrayR _ d) _ gvbOut) kernel
 | Refl <- reprIsSingle @ScalarType @a @Buffer stIn1
 , Refl <- reprIsSingle @ScalarType @b @Buffer stIn2
 , Refl <- reprIsSingle @ScalarType @c @Buffer stIn3
 , Refl <- reprIsSingle @ScalarType @d @Buffer stOut
 = kernel (groundToBase a gvbIn1) (groundToBase b gvbIn2) (groundToBase c gvbIn3) (groundToBase d gvbOut)
compileTernaryOperation1 _ _ _ _ _ _ _ = error "impossible"

groundToBase :: TypeR a -> GroundVars env (Buffer a) -> BaseVar env (Buffer a)
groundToBase _ (TupRsingle (Var groundR idx)) = Var (BaseRground groundR) idx

groundsToBase :: TypeR a -> GroundVars env (Buffers a) -> BaseVars env (Buffers a)
groundsToBase _ TupRunit                              = TupRunit
groundsToBase t@(TupRsingle (st :: ScalarType e)) gvb
  | Refl <- reprIsSingle @ScalarType @e @Buffer st    = TupRsingle (groundToBase t gvb)
groundsToBase (TupRpair t1 t2) (TupRpair gvb1 gvb2)   = TupRpair (groundsToBase t1 gvb1) (groundsToBase t2 gvb2)
groundsToBase _ _                                     = error "impossible"